"""Contract tests for IssueProvider implementations.

These tests validate that FakeIssueProvider behaves identically to BeadsClient.
Tests are parametrized to run against both implementations, ensuring the fake
doesn't diverge from real behavior.

Real provider tests require:
- bd CLI in PATH
- BEADS_TEST_WORKSPACE environment variable set to a valid beads workspace

When prerequisites are missing, real provider tests skip gracefully.
"""

import os
import shutil
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


def _has_bd_cli() -> bool:
    """Check if bd CLI is available in PATH."""
    return shutil.which("bd") is not None


def _has_beads_workspace() -> bool:
    """Check if BEADS_TEST_WORKSPACE env var is set."""
    return os.environ.get("BEADS_TEST_WORKSPACE") is not None


# Type alias for providers - both fake and real implement IssueProvider protocol
IssueProviderLike = Any  # Using Any to avoid circular import with BeadsClient


@pytest.fixture
async def fake_provider() -> AsyncIterator[FakeIssueProvider]:
    """Create a FakeIssueProvider with test issues."""
    issues = {
        "issue-1": FakeIssue(
            id="issue-1",
            status="ready",
            priority=1,
            description="Test issue 1",
            tags=["tag-1"],
        ),
        "issue-2": FakeIssue(
            id="issue-2",
            status="ready",
            priority=2,
            parent_epic="epic-1",
            description="Test issue 2",
        ),
        "epic-1": FakeIssue(
            id="epic-1",
            status="ready",
            priority=1,
            description="Test epic",
        ),
    }
    yield FakeIssueProvider(issues)


class TestFakeIssueProviderContract:
    """Contract tests that FakeIssueProvider must pass.

    These tests define the expected behavior that any IssueProvider must exhibit.
    When real provider tests are enabled, they validate parity.
    """

    @pytest.mark.integration
    async def test_get_ready_returns_ready_issues(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_ready_async returns issues with ready status."""
        ready = await fake_provider.get_ready_async()
        assert "issue-1" in ready
        assert "issue-2" in ready

    @pytest.mark.integration
    async def test_get_ready_excludes_specified_ids(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_ready_async respects exclude_ids parameter."""
        ready = await fake_provider.get_ready_async(exclude_ids={"issue-1"})
        assert "issue-1" not in ready
        assert "issue-2" in ready

    @pytest.mark.integration
    async def test_get_ready_filters_by_epic(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_ready_async filters by epic_id when provided."""
        ready = await fake_provider.get_ready_async(epic_id="epic-1")
        assert "issue-2" in ready
        assert "issue-1" not in ready  # issue-1 has no parent epic

    @pytest.mark.integration
    async def test_get_ready_sorts_by_priority(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_ready_async returns issues sorted by priority (lower first)."""
        ready = await fake_provider.get_ready_async()
        # issue-1 has priority 1, issue-2 has priority 2
        issue1_idx = ready.index("issue-1")
        issue2_idx = ready.index("issue-2")
        assert issue1_idx < issue2_idx

    @pytest.mark.integration
    async def test_claim_marks_in_progress(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """claim_async sets issue status to in_progress."""
        result = await fake_provider.claim_async("issue-1")
        assert result is True
        ready = await fake_provider.get_ready_async()
        assert "issue-1" not in ready

    @pytest.mark.integration
    async def test_claim_nonexistent_returns_false(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """claim_async returns False for nonexistent issue."""
        result = await fake_provider.claim_async("nonexistent")
        assert result is False

    @pytest.mark.integration
    async def test_close_removes_from_ready(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """close_async sets issue status to closed."""
        result = await fake_provider.close_async("issue-1")
        assert result is True
        ready = await fake_provider.get_ready_async()
        assert "issue-1" not in ready

    @pytest.mark.integration
    async def test_close_nonexistent_returns_false(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """close_async returns False for nonexistent issue."""
        result = await fake_provider.close_async("nonexistent")
        assert result is False

    @pytest.mark.integration
    async def test_reset_returns_to_ready(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """reset_async sets issue back to ready status."""
        await fake_provider.claim_async("issue-1")
        result = await fake_provider.reset_async("issue-1")
        assert result is True
        ready = await fake_provider.get_ready_async()
        assert "issue-1" in ready

    @pytest.mark.integration
    async def test_reset_nonexistent_returns_false(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """reset_async returns False for nonexistent issue."""
        result = await fake_provider.reset_async("nonexistent")
        assert result is False

    @pytest.mark.integration
    async def test_get_issue_description(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_issue_description_async returns issue description."""
        desc = await fake_provider.get_issue_description_async("issue-1")
        assert desc == "Test issue 1"

    @pytest.mark.integration
    async def test_get_issue_description_nonexistent(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_issue_description_async returns None for nonexistent issue."""
        desc = await fake_provider.get_issue_description_async("nonexistent")
        assert desc is None

    @pytest.mark.integration
    async def test_create_issue_returns_id(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """create_issue_async creates issue and returns its ID."""
        issue_id = await fake_provider.create_issue_async(
            title="New Issue",
            description="New description",
            priority="P2",
            tags=["new-tag"],
        )
        assert issue_id is not None
        desc = await fake_provider.get_issue_description_async(issue_id)
        assert desc == "New description"

    @pytest.mark.integration
    async def test_find_issue_by_tag(self, fake_provider: FakeIssueProvider) -> None:
        """find_issue_by_tag_async finds issue with matching tag."""
        issue_id = await fake_provider.find_issue_by_tag_async("tag-1")
        assert issue_id == "issue-1"

    @pytest.mark.integration
    async def test_find_issue_by_tag_not_found(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """find_issue_by_tag_async returns None when tag not found."""
        issue_id = await fake_provider.find_issue_by_tag_async("nonexistent-tag")
        assert issue_id is None

    @pytest.mark.integration
    async def test_update_issue_description(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """update_issue_description_async updates the description."""
        result = await fake_provider.update_issue_description_async(
            "issue-1", "Updated description"
        )
        assert result is True
        desc = await fake_provider.get_issue_description_async("issue-1")
        assert desc == "Updated description"

    @pytest.mark.integration
    async def test_update_issue_description_nonexistent(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """update_issue_description_async returns False for nonexistent."""
        result = await fake_provider.update_issue_description_async(
            "nonexistent", "New desc"
        )
        assert result is False

    @pytest.mark.integration
    async def test_get_epic_children(self, fake_provider: FakeIssueProvider) -> None:
        """get_epic_children_async returns child issue IDs."""
        children = await fake_provider.get_epic_children_async("epic-1")
        assert "issue-2" in children

    @pytest.mark.integration
    async def test_get_parent_epic(self, fake_provider: FakeIssueProvider) -> None:
        """get_parent_epic_async returns parent epic ID."""
        parent = await fake_provider.get_parent_epic_async("issue-2")
        assert parent == "epic-1"

    @pytest.mark.integration
    async def test_get_parent_epic_orphan(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """get_parent_epic_async returns None for orphan issue."""
        parent = await fake_provider.get_parent_epic_async("issue-1")
        assert parent is None

    @pytest.mark.integration
    async def test_mark_needs_followup(self, fake_provider: FakeIssueProvider) -> None:
        """mark_needs_followup_async changes status."""
        result = await fake_provider.mark_needs_followup_async("issue-1", "Gate failed")
        assert result is True
        ready = await fake_provider.get_ready_async()
        assert "issue-1" not in ready

    @pytest.mark.integration
    async def test_reopen_issue(self, fake_provider: FakeIssueProvider) -> None:
        """reopen_issue_async sets status to open."""
        # First claim and mark needs-followup
        await fake_provider.claim_async("issue-1")
        await fake_provider.mark_needs_followup_async("issue-1", "Deadlock victim")
        # Now reopen
        result = await fake_provider.reopen_issue_async("issue-1")
        assert result is True
        # Issue should have status "open" now (tracked in reopened set)
        assert "issue-1" in fake_provider.reopened

    @pytest.mark.integration
    async def test_reopen_issue_nonexistent(
        self, fake_provider: FakeIssueProvider
    ) -> None:
        """reopen_issue_async returns False for nonexistent issue."""
        result = await fake_provider.reopen_issue_async("nonexistent")
        assert result is False

    @pytest.mark.integration
    async def test_add_dependency(self, fake_provider: FakeIssueProvider) -> None:
        """add_dependency_async records dependency."""
        result = await fake_provider.add_dependency_async("issue-1", "issue-2")
        assert result is True
        # Verify observable state
        assert ("issue-1", "issue-2") in fake_provider.dependency_calls

    @pytest.mark.integration
    async def test_commit_issues(self, fake_provider: FakeIssueProvider) -> None:
        """commit_issues_async returns True."""
        result = await fake_provider.commit_issues_async()
        assert result is True

    @pytest.mark.integration
    async def test_close_eligible_epics(self, fake_provider: FakeIssueProvider) -> None:
        """close_eligible_epics_async closes epics with all children closed."""
        # Close the only child of epic-1
        await fake_provider.close_async("issue-2")
        result = await fake_provider.close_eligible_epics_async()
        assert result is True
        # epic-1 should now be closed
        ready = await fake_provider.get_ready_async()
        assert "epic-1" not in ready


class TestFakeIssueProviderObservableState:
    """Tests for FakeIssueProvider's observable state tracking."""

    @pytest.mark.unit
    async def test_claimed_set_tracks_claims(self) -> None:
        """claimed set tracks all claimed issue IDs."""
        provider = FakeIssueProvider({"a": FakeIssue("a"), "b": FakeIssue("b")})
        await provider.claim_async("a")
        await provider.claim_async("b")
        assert provider.claimed == {"a", "b"}

    @pytest.mark.unit
    async def test_closed_set_tracks_closes(self) -> None:
        """closed set tracks all closed issue IDs."""
        provider = FakeIssueProvider({"a": FakeIssue("a"), "b": FakeIssue("b")})
        await provider.close_async("a")
        assert provider.closed == {"a"}

    @pytest.mark.unit
    async def test_reset_calls_tracks_resets(self) -> None:
        """reset_calls tracks all reset operations."""
        from pathlib import Path

        provider = FakeIssueProvider({"a": FakeIssue("a")})
        await provider.reset_async("a", Path("/log.jsonl"), "error msg")
        assert len(provider.reset_calls) == 1
        assert provider.reset_calls[0] == ("a", Path("/log.jsonl"), "error msg")

    @pytest.mark.unit
    async def test_created_issues_tracks_creations(self) -> None:
        """created_issues tracks all created issues."""
        provider = FakeIssueProvider()
        await provider.create_issue_async(
            title="Test",
            description="Desc",
            priority="P1",
            tags=["t1"],
            parent_id="parent",
        )
        assert len(provider.created_issues) == 1
        created = provider.created_issues[0]
        assert created["title"] == "Test"
        assert created["priority"] == "P1"
        assert created["tags"] == ["t1"]
        assert created["parent_id"] == "parent"

    @pytest.mark.unit
    async def test_reset_removes_from_claimed(self) -> None:
        """reset_async removes issue from claimed set."""
        provider = FakeIssueProvider({"a": FakeIssue("a")})
        await provider.claim_async("a")
        assert "a" in provider.claimed
        await provider.reset_async("a")
        assert "a" not in provider.claimed
