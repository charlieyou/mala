"""Unit tests for compute_dry_run_outcome.

Exercise the dry-run orchestration helper without Typer or the CLI. The
helper resolves the ready-issue list the run would process and returns a
typed outcome; the CLI is responsible for formatting/exit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.core.models import OrderPreference
from src.orchestration.dry_run import DryRunOutcome, compute_dry_run_outcome

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


pytestmark = pytest.mark.unit


class FakeIssueProvider:
    """In-memory IssueProvider stand-in capturing the dry-run query."""

    def __init__(
        self,
        repo_path: Path,
        issues: list[dict[str, object]],
    ) -> None:
        self.repo_path = repo_path
        self._issues = issues
        self.last_kwargs: dict[str, object] | None = None

    async def get_ready_issues_async(
        self,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        self.last_kwargs = kwargs
        return self._issues


def _factory(
    issues: list[dict[str, object]] | None = None,
    capture: list[FakeIssueProvider] | None = None,
) -> Callable[[Path], FakeIssueProvider]:
    """Return a factory that yields a fresh FakeIssueProvider per call."""
    issues_to_return = issues if issues is not None else []

    def make(repo_path: Path) -> FakeIssueProvider:
        provider = FakeIssueProvider(repo_path, issues_to_return)
        if capture is not None:
            capture.append(provider)
        return provider

    return make


class TestDryRunOutcomeShape:
    """Outcome dataclass shape."""

    def test_outcome_carries_issues_and_focus(self) -> None:
        outcome = DryRunOutcome(issues=[{"id": "x"}], focus=True)
        assert outcome.issues == [{"id": "x"}]
        assert outcome.focus is True


class TestFocusDerivation:
    """``focus`` is derived from ``order_preference``."""

    @pytest.mark.parametrize(
        "order_preference,expected_focus",
        [
            (OrderPreference.FOCUS, True),
            (OrderPreference.EPIC_PRIORITY, True),
            (OrderPreference.ISSUE_PRIORITY, False),
            (OrderPreference.INPUT, False),
        ],
    )
    def test_focus_matches_order_preference(
        self,
        order_preference: OrderPreference,
        expected_focus: bool,
        tmp_path: Path,
    ) -> None:
        outcome = compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=None,
            orphans_only=False,
            include_wip=False,
            order_preference=order_preference,
            issue_provider_factory=_factory(),
        )
        assert outcome.focus is expected_focus


class TestProviderQuery:
    """compute_dry_run_outcome forwards all filtering params to the provider."""

    def test_forwards_ids_resume_and_focus(self, tmp_path: Path) -> None:
        captured: list[FakeIssueProvider] = []
        compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=["id-1", "id-2"],
            orphans_only=False,
            include_wip=True,
            order_preference=OrderPreference.EPIC_PRIORITY,
            issue_provider_factory=_factory(capture=captured),
        )
        assert len(captured) == 1
        provider = captured[0]
        assert provider.repo_path == tmp_path
        assert provider.last_kwargs is not None
        assert provider.last_kwargs["only_ids"] == ["id-1", "id-2"]
        assert provider.last_kwargs["include_wip"] is True
        assert provider.last_kwargs["focus"] is True
        assert provider.last_kwargs["epic_id"] is None
        assert provider.last_kwargs["orphans_only"] is False
        assert provider.last_kwargs["order_preference"] == OrderPreference.EPIC_PRIORITY

    def test_forwards_epic_filter(self, tmp_path: Path) -> None:
        captured: list[FakeIssueProvider] = []
        compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id="epic-42",
            only_ids=None,
            orphans_only=False,
            include_wip=False,
            order_preference=OrderPreference.FOCUS,
            issue_provider_factory=_factory(capture=captured),
        )
        provider = captured[0]
        assert provider.last_kwargs is not None
        assert provider.last_kwargs["epic_id"] == "epic-42"
        assert provider.last_kwargs["only_ids"] is None
        assert provider.last_kwargs["orphans_only"] is False

    def test_forwards_orphans_only(self, tmp_path: Path) -> None:
        captured: list[FakeIssueProvider] = []
        compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=None,
            orphans_only=True,
            include_wip=False,
            order_preference=OrderPreference.ISSUE_PRIORITY,
            issue_provider_factory=_factory(capture=captured),
        )
        provider = captured[0]
        assert provider.last_kwargs is not None
        assert provider.last_kwargs["orphans_only"] is True
        assert provider.last_kwargs["focus"] is False


class TestIssuesReturned:
    """Resolved issues land on the outcome unchanged."""

    def test_returns_provider_issues(self, tmp_path: Path) -> None:
        issues: list[dict[str, object]] = [
            {
                "id": "task-1",
                "title": "First",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-1",
            },
            {
                "id": "task-2",
                "title": "Second",
                "priority": 2,
                "status": "in_progress",
                "parent_epic": None,
            },
        ]
        outcome = compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=None,
            orphans_only=False,
            include_wip=True,
            order_preference=OrderPreference.EPIC_PRIORITY,
            issue_provider_factory=_factory(issues=issues),
        )
        assert outcome.issues == issues

    def test_empty_result_returns_empty_list(self, tmp_path: Path) -> None:
        outcome = compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=None,
            orphans_only=False,
            include_wip=False,
            order_preference=OrderPreference.EPIC_PRIORITY,
            issue_provider_factory=_factory(issues=[]),
        )
        assert outcome.issues == []
        assert outcome.focus is True


class TestDefaultFactory:
    """Default factory resolves through src.orchestration.factory."""

    def test_default_uses_module_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatching the factory module is observable via the default path."""
        import src.orchestration.factory as factory_module

        captured: list[FakeIssueProvider] = []

        def fake_create_issue_provider(
            repo_path: Path, log_warning: object = None
        ) -> FakeIssueProvider:
            provider = FakeIssueProvider(repo_path, [])
            captured.append(provider)
            return provider

        monkeypatch.setattr(
            factory_module, "create_issue_provider", fake_create_issue_provider
        )

        outcome = compute_dry_run_outcome(
            repo_path=tmp_path,
            epic_id=None,
            only_ids=None,
            orphans_only=False,
            include_wip=False,
            order_preference=OrderPreference.EPIC_PRIORITY,
        )
        assert outcome.issues == []
        assert len(captured) == 1
        assert captured[0].repo_path == tmp_path
