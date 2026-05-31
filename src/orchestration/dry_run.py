"""Dry-run orchestration helper.

Moves the dry-run flow out of the Typer CLI so the CLI keeps argument
parsing and display formatting only. ``compute_dry_run_outcome`` loads
the ready-issue list the run would process and returns a typed value
object the CLI then renders.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from src.core.models import OrderPreference
from src.orchestration import factory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _DryRunIssueProvider(Protocol):
    """Subset of ``IssueProvider`` used by the dry-run helper.

    ``get_ready_issues_async`` returns the full issue dicts (id, title,
    priority, status, parent_epic) needed for dry-run display. It lives on
    ``BeadsClient`` rather than the wider ``IssueProvider`` protocol so the
    helper declares the narrow contract it actually depends on.
    """

    async def get_ready_issues_async(
        self,
        *,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class DryRunOutcome:
    """Result of computing the dry-run task list.

    Attributes:
        issues: Issues the run would process, with id/title/priority/status/parent_epic.
        focus: True when the CLI should render with epic grouping.
    """

    issues: list[dict[str, object]]
    focus: bool


def _resolve_focus(order_preference: OrderPreference) -> bool:
    return order_preference in (
        OrderPreference.FOCUS,
        OrderPreference.EPIC_PRIORITY,
    )


def compute_dry_run_outcome(
    repo_path: Path,
    *,
    epic_id: str | None,
    only_ids: list[str] | None,
    orphans_only: bool,
    include_wip: bool,
    order_preference: OrderPreference,
    prioritize_wip: bool = False,
    issue_provider_factory: Callable[[Path], _DryRunIssueProvider] | None = None,
) -> DryRunOutcome:
    """Compute the dry-run outcome for the requested run scope.

    Args:
        repo_path: Repository path the run targets.
        epic_id: Optional epic filter (from ``--scope epic:<id>``).
        only_ids: Optional explicit ID list (from ``--scope ids:<id,...>``).
        orphans_only: Whether to restrict to orphan issues.
        include_wip: Include in-progress issues (``--resume``).
        prioritize_wip: Move in-progress issues before ready issues.
        order_preference: Issue ordering preference.
        issue_provider_factory: Optional factory override; defaults to
            ``src.orchestration.factory.create_issue_provider``.

    Returns:
        A ``DryRunOutcome`` carrying the resolved issues and display focus.
    """
    focus = _resolve_focus(order_preference)

    async def _load() -> list[dict[str, object]]:
        if issue_provider_factory is not None:
            provider = issue_provider_factory(repo_path)
        else:
            provider = cast(
                "_DryRunIssueProvider", factory.create_issue_provider(repo_path)
            )
        issues = await provider.get_ready_issues_async(
            epic_id=epic_id,
            only_ids=only_ids,
            include_wip=include_wip,
            focus=focus,
            orphans_only=orphans_only,
            order_preference=order_preference,
        )
        if include_wip and prioritize_wip:
            return sorted(
                issues,
                key=lambda issue: 0 if issue.get("status") == "in_progress" else 1,
            )
        return issues

    issues = asyncio.run(_load())
    return DryRunOutcome(issues=issues, focus=focus)
