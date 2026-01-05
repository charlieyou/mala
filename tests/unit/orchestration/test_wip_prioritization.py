"""Unit tests for --wip flag prioritization logic and focus mode.

Tests use FakeIssueProvider for behavioral assertions on issue ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator

from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


class TestPrioritizeWipFlag:
    """Test --wip flag prioritization via FakeIssueProvider.get_ready_async."""

    @pytest.mark.asyncio
    async def test_prioritize_wip_sorts_in_progress_first(self) -> None:
        """When prioritize_wip=True, in_progress issues should come before open."""
        fake_issues = FakeIssueProvider(
            {
                "open-1": FakeIssue(id="open-1", priority=1, status="open"),
                "wip-1": FakeIssue(id="wip-1", priority=2, status="in_progress"),
                "open-2": FakeIssue(id="open-2", priority=3, status="open"),
            }
        )

        result = await fake_issues.get_ready_async(prioritize_wip=True)

        # in_progress should come first, then open sorted by priority
        assert result == ["wip-1", "open-1", "open-2"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_false_uses_priority_only(self) -> None:
        """When prioritize_wip=False (default), sort by priority only."""
        fake_issues = FakeIssueProvider(
            {
                "wip-1": FakeIssue(id="wip-1", priority=3, status="in_progress"),
                "open-1": FakeIssue(id="open-1", priority=1, status="open"),
                "open-2": FakeIssue(id="open-2", priority=2, status="open"),
            }
        )

        result = await fake_issues.get_ready_async(prioritize_wip=False)

        # Without prioritize_wip, in_progress is NOT included (only open status)
        assert result == ["open-1", "open-2"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_respects_priority_within_status(self) -> None:
        """Within each status group, issues should still be sorted by priority."""
        fake_issues = FakeIssueProvider(
            {
                "wip-high": FakeIssue(id="wip-high", priority=1, status="in_progress"),
                "wip-low": FakeIssue(id="wip-low", priority=3, status="in_progress"),
                "open-high": FakeIssue(id="open-high", priority=1, status="open"),
                "open-low": FakeIssue(id="open-low", priority=2, status="open"),
            }
        )

        result = await fake_issues.get_ready_async(prioritize_wip=True)

        # in_progress first (sorted by priority), then open (sorted by priority)
        assert result == ["wip-high", "wip-low", "open-high", "open-low"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_default_is_false(self) -> None:
        """Default behavior should not prioritize in_progress issues."""
        fake_issues = FakeIssueProvider(
            {
                "wip-1": FakeIssue(id="wip-1", priority=3, status="in_progress"),
                "open-1": FakeIssue(id="open-1", priority=1, status="open"),
            }
        )

        # Don't pass prioritize_wip - should default to False
        result = await fake_issues.get_ready_async()

        # Without prioritize_wip=True, only open issues are returned
        assert result == ["open-1"]


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
    """Test focus mode epic-grouped ordering via FakeIssueProvider."""

    @pytest.mark.asyncio
    async def test_focus_groups_tasks_by_epic(self) -> None:
        """When focus=True, tasks should be grouped by parent epic."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=2,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-b1": FakeIssue(
                    id="task-b1",
                    priority=1,
                    status="open",
                    parent_epic="epic-b",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-a2": FakeIssue(
                    id="task-a2",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=True)

        # epic-a has min_priority=1 (task-a2), epic-b has min_priority=1 (task-b1)
        # Within epic-a: task-a2 (P1) before task-a1 (P2)
        assert result == ["task-a2", "task-a1", "task-b1"]

    @pytest.mark.asyncio
    async def test_focus_false_uses_priority_only(self) -> None:
        """When focus=False, tasks should be interleaved by priority."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=2,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-b1": FakeIssue(
                    id="task-b1",
                    priority=1,
                    status="open",
                    parent_epic="epic-b",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-a2": FakeIssue(
                    id="task-a2",
                    priority=3,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=False)

        # Should sort by priority only: P1, P2, P3
        assert result == ["task-b1", "task-a1", "task-a2"]

    @pytest.mark.asyncio
    async def test_focus_orphan_tasks_form_virtual_group(self) -> None:
        """Orphan tasks (no parent epic) should form their own virtual group."""
        fake_issues = FakeIssueProvider(
            {
                "orphan-1": FakeIssue(
                    id="orphan-1",
                    priority=3,
                    status="open",
                    parent_epic=None,
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=2,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "orphan-2": FakeIssue(
                    id="orphan-2",
                    priority=1,
                    status="open",
                    parent_epic=None,
                    updated_at="2025-01-01T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=True)

        # Orphan group has min_priority=1 (orphan-2), epic-a has min_priority=2
        # Orphan group comes first, sorted by priority: orphan-2 (P1), orphan-1 (P3)
        # Then epic-a group: task-a1 (P2)
        assert result == ["orphan-2", "orphan-1", "task-a1"]

    @pytest.mark.asyncio
    async def test_focus_tiebreaker_uses_updated_at(self) -> None:
        """Equal-priority groups should be ordered by most recently updated."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-b1": FakeIssue(
                    id="task-b1",
                    priority=1,
                    status="open",
                    parent_epic="epic-b",
                    updated_at="2025-01-02T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=True)

        # Both epics have P1, but epic-b updated more recently (2025-01-02)
        # So epic-b should come first
        assert result == ["task-b1", "task-a1"]

    @pytest.mark.asyncio
    async def test_focus_with_prioritize_wip(self) -> None:
        """WIP prioritization should still move in_progress to front globally."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-b1": FakeIssue(
                    id="task-b1",
                    priority=2,
                    status="open",
                    parent_epic="epic-b",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-b2": FakeIssue(
                    id="task-b2",
                    priority=3,
                    status="in_progress",
                    parent_epic="epic-b",
                    updated_at="2025-01-01T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=True, prioritize_wip=True)

        # WIP issue task-b2 should come first globally
        # Then remaining tasks grouped by epic:
        # epic-a (P1) before epic-b (P2 remaining after task-b2)
        assert result[0] == "task-b2"
        # The rest should be epic-grouped: epic-a (task-a1) then epic-b (task-b1)
        assert result == ["task-b2", "task-a1", "task-b1"]

    @pytest.mark.asyncio
    async def test_focus_default_is_true(self) -> None:
        """Default focus should be True."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=2,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-a2": FakeIssue(
                    id="task-a2",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
            }
        )

        # Don't pass focus - should default to True and use epic grouping
        result = await fake_issues.get_ready_async()

        # Tasks grouped by epic, sorted by priority within
        assert result == ["task-a2", "task-a1"]

    @pytest.mark.asyncio
    async def test_focus_within_group_sorts_by_priority_then_updated(self) -> None:
        """Within an epic group, tasks should sort by priority then updated DESC."""
        fake_issues = FakeIssueProvider(
            {
                "task-a1": FakeIssue(
                    id="task-a1",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-01T10:00:00Z",
                ),
                "task-a2": FakeIssue(
                    id="task-a2",
                    priority=1,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-02T10:00:00Z",
                ),
                "task-a3": FakeIssue(
                    id="task-a3",
                    priority=2,
                    status="open",
                    parent_epic="epic-a",
                    updated_at="2025-01-03T10:00:00Z",
                ),
            }
        )

        result = await fake_issues.get_ready_async(focus=True)

        # All in same epic, sorted by (priority, updated DESC)
        # P1 tasks first: task-a2 (2025-01-02) before task-a1 (2025-01-01)
        # Then P2: task-a3
        assert result == ["task-a2", "task-a1", "task-a3"]
